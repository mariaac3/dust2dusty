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
    - SALT2mu.exe interface via salt2mu module
    - Likelihood calculation comparing multiple observables
    - Support for parameter splits by mass, redshift, etc.

Usage:
    python DUST2DUST.py --CONFIG IN_DUST2DUST.yml

    Optional flags:
    --SINGLE: Run single likelihood evaluation for testing
    --DOPLOT: Create plots from existing chains
    --DEBUG: Enable verbose output
"""

import os
from collections import defaultdict
from pathlib import Path

from dust2dusty.utils import cmd_exe, normhisttodata, pconv, set_numpy_threads

# Call BEFORE importing numpy
set_numpy_threads(4)

from multiprocessing import Pool, cpu_count, current_process

import emcee
import numpy as np

from dust2dusty.logging import get_logger
from dust2dusty.salt2mu import SALT2mu

# =============================================================================
# GLOBAL VARIABLES & CONSTANTS
# =============================================================================

# Constants
JOBNAME_SALT2mu = "SALT2mu.exe"  # SALT2mu executable name
ncbins = 6  # Number of color bins


# Module-level logger
logger = get_logger()

# Worker-local global variables for multiprocessing
# These are set by _init_worker() for each Pool worker process
_WORKER_REALDATA_SALT2MU_RESULTS = None  # SALT2mu connection for real data
_WORKER_SALT2MU_CONNECTION = None  # SALT2mu connection for simulation
_WORKER_DEBUGFLAG = False  # Debug flag for worker process
_WORKER_INDEX = None  # Worker process index
_CONFIG = None  # Configuration object


# =============================================================================
# PARAMETER CONVERSION HELPERS
# =============================================================================


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
        _CONFIG.inp_params,
        _CONFIG.paramshapesdict,
        _CONFIG.splitdict,
        _CONFIG.DISTRIBUTION_PARAMETERS,
    )  # expanded list of all variables. len is ndim.
    for p in _CONFIG.inp_params:
        thetalist = []
        for n, ep in enumerate(extparams):
            if p in ep:  # for instance, if 'c' is in 'c_l', then this records that position.
                thetalist.append(n)
        thetadict[p] = thetalist
    return thetadict


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
        return theta[lowbound:highbound]


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
    arrlist.append(_CONFIG.DEFAULT_PARAMETER_RANGES[inp])
    if inp in SPLITDICT.keys():
        for s in SPLITDICT[inp].keys():
            arrlist.append(eval(SPLITARR[s]))
    return arrlist


# =============================================================================
# DATA PROCESSING
# =============================================================================


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

    dflow = df.loc[df[f"ibin_{_CONFIG.splitparam}"] == 0]
    dfhigh = df.loc[df[f"ibin_{_CONFIG.splitparam}"] == 1]

    lowNEVT = dflow.NEVT.values
    highNEVT = dfhigh.NEVT.values
    lowrespops = dflow.MURES_SUM.values
    highrespops = dfhigh.MURES_SUM.values

    # Color histogram
    for q in np.unique(df.ibin_c.values):
        cpops.append(np.sum(df.loc[df.ibin_c == q].NEVT))
    cpops = np.array(cpops)

    # x1 (stretch) histogram
    if "ibin_x1" in df.columns:
        for q in np.unique(df.ibin_x1.values):
            x1pops.append(np.sum(df.loc[df.ibin_x1 == q].NEVT))
        x1pops = np.array(x1pops)
    else:
        x1pops = np.array([])

    lowRMS = dflow.STD_ROBUST.values
    highRMS = dfhigh.STD_ROBUST.values

    if RET == "HIST":
        return cpops, x1pops
    elif RET == "ANALYSIS":
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


# =============================================================================
# SALT2MU CONNECTION MANAGEMENT
# =============================================================================


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


def init_salt2mu_worker_connection(config, index, debug=False):
    """
    Initialize connection(s) to SALT2mu.exe subprocess(es).

    Creates SALT2mu connection objects for real data and/or simulation.
    Each connection maintains a persistent subprocess that can be called repeatedly
    with different PDF functions for reweighting.

    Args:
        config: Configuration object with paths and parameters
        index: Integer ID for this connection (used for file naming in parallel/)
        real: If True, also create connection for real data (default: True)
        debug: If True, use OPTMASK=1 to create FITRES files (default: False)

    Returns:
        tuple: (realdata, connection)
               realdata: SALT2mu object for real data, or None if real=False
               connection: SALT2mu object for simulation

    Side effects:
        - Creates temporary files in config.outdir/parallel/ for subprocess I/O
        - Launches SALT2mu.exe subprocess(es)

    OPTMASK values:
        1: Creates FITRES file (used in DEBUG/SINGLE modes)
        2: Creates M0DIF file
        4: Implements randomseed option (default for production)
    """
    OPTMASK = 4
    directory = "worker_files"
    if debug:
        OPTMASK = 1

    OUTDIR = Path(config.outdir)
    sim_data_out = OUTDIR / f"{directory}/{index}_SUBPROCESS_SIM_OUT.DAT"
    sim_data_out.touch()

    maps_out = OUTDIR / f"{directory}/{index}_PYTHONCROSSTALK_OUT.DAT"
    maps_out.touch()

    subprocess_log_sim = OUTDIR / f"{directory}/{index}_SUBPROCESS_LOG_SIM.STDOUT"
    subprocess_log_sim.touch()

    # Generate output table specification (color bins x split parameter bins)
    arg_outtable = f"'c(6,-0.2:0.25)*{config.SPLIT_PARAMETER_FORMATS[config.splitparam]}'"

    # Generate GENPDF variable names from input parameters
    GENPDF_NAMES = generate_genpdf_varnames(config.inp_params, config.splitparam)

    cmd = cmd_exe(JOBNAME_SALT2mu, config.sim_input) + (
        f"SUBPROCESS_VARNAMES_GENPDF={GENPDF_NAMES} "
        f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} "
        f"SUBPROCESS_OPTMASK={OPTMASK} "
        f"SUBPROCESS_SIMREF_FILE={config.simref_file} "
        f"debug_flag=930"
    )

    connection = SALT2mu(cmd, mapsout, sim_data_out, subprocess_log_sim, debug=debug)

    return connection


# =============================================================================
# LIKELIHOOD & PRIOR FUNCTIONS
# =============================================================================


def compute_and_sum_loglikelihoods(inparr, returnall=False, RMS_weight=1):
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
        inparr: Dictionary with [data, sim] pairs for each observable:
                Keys: 'color_hist', 'x1_hist', 'mures_high', 'mures_low',
                      'rms_high', 'rms_low', 'nevt_high', 'nevt_low'
                Each value is [real_data, sim_data]
        returnall: If True, return detailed components (default: False)
        RMS_weight: Weight factor for RMS terms in likelihood (default: 1)

    Returns:
        If returnall is False:
            float: Total log-likelihood (sum of all components)
        If returnall is True:
            tuple: (total_ll, ll_dict, datacount_dict, simcount_dict, poisson_dict)
    """
    ll_dict = defaultdict(float)
    datacount_dict = defaultdict(float)
    simcount_dict = defaultdict(float)
    poisson_dict = defaultdict(float)

    # ========== Parameter likelihood terms ==========
    # Beta (color-luminosity relation)
    logger.debug(
        f"real beta, sim beta, real beta error: {_WORKER_REALDATA_SALT2MU_RESULTS['beta']}, "
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
        datacount_color, simcount_color, poisson_color, ww = normhisttodata(data_color, sim_color)
        ll_dict["color_hist"] = -0.5 * np.sum(
            (datacount_color - simcount_color) ** 2 / poisson_color**2
        )
        datacount_dict["color_hist"] = datacount_color
        simcount_dict["color_hist"] = simcount_color
        poisson_dict["color_hist"] = poisson_color

    # X1 (stretch) histogram
    data_x1, sim_x1 = inparr["x1_hist"]
    if len(data_x1) > 0 and len(sim_x1) > 0:
        datacount_x1, simcount_x1, poisson_x1, ww = normhisttodata(data_x1, sim_x1)
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
        -0.5 * np.sum((data_rms_high - sim_rms_high) ** 2 / poisson_rms_high**2) * RMS_weight
    )
    datacount_dict["rms_high"] = data_rms_high
    simcount_dict["rms_high"] = sim_rms_high
    poisson_dict["rms_high"] = poisson_rms_high

    # Low-mass RMS
    data_rms_low, sim_rms_low = inparr["rms_low"]
    poisson_rms_low = data_rms_low / np.sqrt(2 * nevt_low)
    ll_dict["rms_low"] = (
        -0.5 * np.sum((data_rms_low - sim_rms_low) ** 2 / poisson_rms_low**2) * RMS_weight
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
        logger.warning(f"ll_dict values: {ll_dict}")
        if returnall:
            return ll_dict, datacount_dict, simcount_dict, poisson_dict
        else:
            return -np.inf

    if returnall:
        return sum(ll_dict.values()), ll_dict, datacount_dict, simcount_dict, poisson_dict

    return sum(ll_dict.values())


def log_likelihood(theta, returnall: bool = False, debug: bool = False):
    """
    Calculate log-likelihood for proposed parameter values.

    Core likelihood function for MCMC. For each parameter set:
    1. Writes PDF functions to file via connection.write_generic_PDF()
    2. Calls SALT2mu.exe to reweight simulation with those PDFs
    3. Parses binned output from SALT2mu (color histograms, MURES, RMS by mass)
    4. Compares reweighted simulation to real data via compute_and_sum_loglikelihoods()

    Args:
        theta: Array of parameter values (length = ndim)
        returnall: If True, return detailed likelihood components (default: False)
        debug: Debug flag (default: False)

    Returns:
        float: Log-likelihood value
        If returnall=True: tuple of (ll_dict, datacount_dict, simcount_dict, poisson_dict)
        Returns -inf if MAXPROB > 1.001 (PDF hitting boundary)
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
    sim_vals = dffixer(bindf, "ANALYSIS", False)

    realbindf = _WORKER_REALDATA_SALT2MU_RESULTS["bindf"].dropna()
    real_vals = dffixer(realbindf, "ANALYSIS", True)

    # Build dictionary pairing data and simulation values
    inparr = {key: [real_vals[key], sim_vals[key]] for key in real_vals.keys()}

    logger.debug("Right before calling LL Creator")
    out_result = compute_and_sum_loglikelihoods(inparr, returnall=returnall)

    _WORKER_SALT2MU_CONNECTION.iter += 1
    return out_result


def log_prior(theta):
    """
    Calculate log-prior probability for parameter values.

    Checks if all parameters are within their allowed bounds specified in
    config.parameter_initialization. Uses uniform (flat) priors within bounds,
    returning 0 (log(1)) if all parameters are valid or -inf if any parameter
    is outside its allowed range.

    Args:
        theta: Array of parameter values (length = ndim)

    Returns:
        float: 0.0 if all parameters within bounds, -np.inf otherwise
    """
    thetadict = thetaconverter(theta)
    plist = pconv(
        _CONFIG.inp_params,
        _CONFIG.paramshapesdict,
        _CONFIG.splitdict,
        _CONFIG.DISTRIBUTION_PARAMETERS,
    )
    logger.debug(f"plist: {plist}")

    tlist = False  # if all parameters are good, this remains false
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
                tlist = True

    if tlist:
        return -np.inf
    else:
        return 0


def log_probability(theta):
    """
    Calculate log-probability (posterior) for MCMC sampling.

    Combines log-prior and log-likelihood following Bayes' theorem.
    Must be called after _init_worker has set up the worker state.

    Args:
        theta: Array of parameter values (length = ndim)

    Returns:
        float: Log-posterior probability (log_prior + log_likelihood)
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        logger.debug("WARNING! We returned -inf from small parameters!")
        return -np.inf
    return lp + log_likelihood(theta)


# =============================================================================
# INITIALIZATION & WORKER SETUP
# =============================================================================


def _init_worker(config, realdata_salt2mu_results, debug=False):
    """
    Initializer function for Pool workers.

    Sets up worker-local state by storing the appropriate connection
    for this worker based on its process identity. Called once per
    worker when the Pool is created.

    Args:
        config: Configuration object
        realdata: SALT2mu connection for real data (shared across workers)
        debug: Debug flag
    """
    global \
        _WORKER_REALDATA_SALT2MU_RESULTS, \
        _WORKER_SALT2MU_CONNECTION, \
        _WORKER_DEBUGFLAG, \
        _CONFIG, \
        _WORKER_INDEX
    _WORKER_DEBUGFLAG = debug
    _CONFIG = config

    if debug:
        _WORKER_INDEX = 999
    else:
        _WORKER_INDEX = current_process()._identity[0] - 1
    _WORKER_SALT2MU_CONNECTION = init_salt2mu_worker_connection(
        _CONFIG, _WORKER_INDEX, real=False, debug=debug
    )
    _WORKER_REALDATA_SALT2MU_RESULTS = realdata_salt2mu_results


# =============================================================================
# MAIN MCMC FUNCTION
# =============================================================================


def MCMC(
    config,
    pos,
    nwalkers,
    ndim,
    realdata,
    debug=False,
    max_iterations=100000,
    convergence_check_interval=100,
):
    """
    Run MCMC sampling using emcee ensemble sampler with HDF5 backend and convergence monitoring.

    Uses the emcee HDF5 backend for robust chain storage and monitors convergence
    via integrated autocorrelation time. Sampling stops when chains are sufficiently
    long relative to autocorrelation time and tau estimates have stabilized.

    Args:
        config: Configuration object with parameters and paths
        pos: Initial walker positions array of shape (nwalkers, ndim)
        nwalkers: Number of MCMC walkers
        ndim: Number of parameters (dimensions)
        realdata: SALT2mu object containing real data fit results
        debug: If True, run in debug mode (default: False)
        max_iterations: Maximum number of iterations before stopping (default: 100000)
        convergence_check_interval: Check convergence every N steps (default: 100)

    Returns:
        emcee.EnsembleSampler: The sampler object with chain results

    Convergence criteria (from emcee documentation):
        1. Chain length > 100 * tau (autocorrelation time)
        2. Tau estimate changed by < 1% since last check

    Side effects:
        - Saves chains to HDF5 file: {outdir}/chains/{data_input}-chains.h5
        - Saves autocorrelation history to: {outdir}/chains/{data_input}-autocorr.npz
        - Saves thinned samples to: {outdir}/chains/{data_input}-samples_thinned.npz
    """
    # Set up HDF5 backend for robust chain storage
    chain_filename = (
        config.outdir + "chains/" + config.data_input.split(".")[0].split("/")[-1] + "-chains.h5"
    )
    backend = emcee.backends.HDFBackend(chain_filename)
    backend.reset(nwalkers, ndim)
    logger.debug(f"Chain storage initialized: {chain_filename}")

    # Track autocorrelation time history
    autocorr_history = np.empty(max_iterations // convergence_check_interval)
    autocorr_index = 0
    old_tau = np.inf

    with Pool(nwalkers, initializer=_init_worker, initargs=(config, realdata, debug)) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend)

        logger.debug(
            f"Starting MCMC with {cpu_count()} CPUs, {nwalkers} walkers, {ndim} dimensions"
        )
        logger.debug(
            f"Max iterations: {max_iterations}, convergence check every {convergence_check_interval} steps"
        )
        logger.debug("=" * 60)

        # Run with convergence monitoring
        for sample in sampler.sample(pos, iterations=max_iterations, progress=True):
            # Only check convergence every N steps
            if sampler.iteration % convergence_check_interval:
                continue

            # Compute autocorrelation time
            try:
                tau = sampler.get_autocorr_time(tol=0)
                autocorr_history[autocorr_index] = np.mean(tau)
                autocorr_index += 1

                # Check convergence criteria
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

                logger.debug(f"\nIteration {sampler.iteration}:")
                logger.debug(f"  Mean tau: {np.mean(tau):.1f}")
                logger.debug(f"  Min tau:  {np.min(tau):.1f}")
                logger.debug(f"  Max tau:  {np.max(tau):.1f}")
                logger.debug(
                    f"  Chain/tau ratio: {sampler.iteration / np.max(tau):.1f} (need > 100)"
                )
                if np.isfinite(old_tau).all():
                    tau_change = np.max(np.abs(old_tau - tau) / tau) * 100
                    logger.debug(f"  Tau change: {tau_change:.2f}% (need < 1%)")

                if converged:
                    logger.debug("\n" + "=" * 60)
                    logger.debug("CONVERGENCE ACHIEVED!")
                    logger.debug(f"  Final iteration: {sampler.iteration}")
                    logger.debug(f"  Final mean tau: {np.mean(tau):.1f}")
                    logger.debug("=" * 60)
                    break

                old_tau = tau

            except emcee.autocorr.AutocorrError:
                logger.debug(f"\nIteration {sampler.iteration}: Chain too short for tau estimate")

        # Save autocorrelation history
        autocorr_filename = (
            config.outdir
            + "chains/"
            + config.data_input.split(".")[0].split("/")[-1]
            + "-autocorr.npz"
        )
        np.savez(autocorr_filename, autocorr=autocorr_history[:autocorr_index])
        logger.debug(f"Autocorrelation history saved to: {autocorr_filename}")

        # Report final statistics
        logger.debug("\n" + "=" * 60)
        logger.debug("MCMC COMPLETE")
        logger.debug("=" * 60)
        try:
            tau = sampler.get_autocorr_time()
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
            logger.debug(f"Final autocorrelation time: {tau}")
            logger.debug(f"Recommended burn-in: {burnin} steps")
            logger.debug(f"Recommended thinning: {thin} steps")
            logger.debug(f"Effective samples: ~{sampler.iteration * nwalkers / np.mean(tau):.0f}")

            # Get flattened samples with burn-in and thinning applied
            flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
            logger.debug(f"Shape of thinned samples: {flat_samples.shape}")

            # Save thinned samples for convenience
            thinned_filename = (
                config.outdir
                + "chains/"
                + config.data_input.split(".")[0].split("/")[-1]
                + "-samples_thinned.npz"
            )
            np.savez(thinned_filename, samples=flat_samples, tau=tau, burnin=burnin, thin=thin)
            logger.debug(f"Thinned samples saved to: {thinned_filename}")

        except emcee.autocorr.AutocorrError:
            logger.warning("Could not compute final autocorrelation time.")
            logger.warning("Chain may be too short for reliable estimates.")
            logger.warning("Consider running longer or checking for convergence issues.")

    return sampler
