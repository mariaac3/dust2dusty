"""
Likelihood and worker initialization for DUST2DUSTY MCMC sampling.

This module implements the core likelihood evaluation logic and per-process
worker state management used by both emcee and nautilus samplers.

Each MPI worker (or the single serial process) stores its own SALT2mu
subprocess connection and the shared real-data results in module-level
globals set by `_init_worker()`.  The MCMC sampler calls `log_probability()`
which internally applies de-logging, evaluates the prior, runs one SALT2mu
iteration, and returns log-posterior.

Key Functions:
    _init_worker: Initialize per-worker global state (SALT2mu connection, config).
    log_probability: Full log-posterior for MCMC (prior + likelihood).
    log_likelihood: Run SALT2mu and compute chi-squared likelihood.
    log_prior: Uniform prior — returns 0 within bounds, -inf outside.
    cleanup_worker: Gracefully shut down this worker's SALT2mu subprocess.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from dust2dusty.log import add_file_handler, get_logger
from dust2dusty.salt2mu import SALT2mu
from dust2dusty.utils import (
    cmd_salt2mu_exe,
    norm_hist_to_data,
)

if TYPE_CHECKING:
    from dust2dusty.cli import Config

# =============================================================================
# GLOBAL VARIABLES & CONSTANTS
# =============================================================================

# Module-level logger
logger: logging.Logger = get_logger()

# Worker-local global variables for multiprocessing
# These are set by _init_worker() for each Pool worker process
_WORKER_REALDATA_SALT2MU_RESULTS: dict[str, Any] | None = None
_WORKER_SALT2MU_CONNECTION: SALT2mu | None = None
_WORKER_DEBUGFLAG: bool = False
_WORKER_INDEX: int | None = None
_CONFIG: Config | None = None
_LIKELIHOOD_PARAMETERS: dict[str, Any] | None = None

# =============================================================================
# SALT2MU CONNECTION MANAGEMENT
# =============================================================================


def _generate_genpdf_varnames(fitted_params: list[str], split_par: str) -> str:
    """
    Generate SUBPROCESS_VARNAMES_GENPDF string for SALT2mu.

    Builds the comma-separated list of variable names that should be included
    in the GENPDF output file for SNANA simulations. Translates internal
    parameter names to SALT2mu column names using PARAM_TO_SALT2MU mapping.

    Args:
        fitted_params: List of parameter names being fit (e.g., ['c', 'RV', 'x1']).
        split_par: List of split parameter names (e.g., ['HOST_LOGMASS']).

    Returns:
        Comma-separated SALT2mu variable names.
        Example: 'SIM_c,HOST_LOGMASS,SIM_RV,SIM_x1,SIM_ZCMB,SIM_beta'

    Note:
        Always includes SIM_ZCMB and SIM_beta even if not in fitted_params,
        as these are required for SALT2mu output.
    """
    varnames: list[str] = []

    # Add parameter variables in SALT2mu format
    for param in fitted_params:
        if param in SALT2mu._PARAM_TO_SALT2MU:
            salt2mu_name = SALT2mu._PARAM_TO_SALT2MU[param]
            if salt2mu_name not in varnames:
                varnames.append(salt2mu_name)

    # Add split parameter if not already included
    for split_p in split_par:
        varnames.insert(-1, split_p)

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


def _init_salt2mu_worker_connection() -> SALT2mu:
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
        - Creates temporary files in config.outdir/worker_salt2mu_files/ for subprocess I/O
        - Launches SALT2mu.exe subprocess

    Note:
        OPTMASK values:
        - 1: Creates FITRES file (used in DEBUG modes)
        - 2: Creates M0DIF file
        - 4: Implements randomseed option (default for production)
    """
    salt2mu_outdir = _CONFIG.OUTPUT_DIR / "worker_salt2mu_files"

    subprocess_salt2mu_res = salt2mu_outdir / f"{_WORKER_INDEX:02d}_SUBPROCESS_SALT2MU_RES.DAT"
    subprocess_salt2mu_res.touch()

    genpdf_crosstalk_file = salt2mu_outdir / f"{_WORKER_INDEX:02d}_GENPDF_PYTHONCROSSTALK.DAT"
    genpdf_crosstalk_file.touch()

    subprocess_salt2mu_log = salt2mu_outdir / f"{_WORKER_INDEX:02d}_SUBPROCESS_SALT2MU_LOG.STDOUT"
    subprocess_salt2mu_log.touch()

    log_dir = _CONFIG.OUTPUT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    # Generate output table specification (color bins x split parameter bins)
    arg_outtable = "'c(6,-0.2:0.25)*HOST_LOGMASS(2,0:20)'"
    # Generate GENPDF variable names from input parameters
    genpdf_names = _generate_genpdf_varnames(list(_CONFIG.fitted_params.keys()), _CONFIG.split_pars)

    cmd = cmd_salt2mu_exe(_CONFIG) + (
        f"SUBPROCESS_VARNAMES_GENPDF={genpdf_names} "
        f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} "
        f"SUBPROCESS_OPTMASK=4 "
        f"SUBPROCESS_SIMREF_FILE={_CONFIG.simref_file} "
    )
    connection = SALT2mu(
        cmd,
        genpdf_crosstalk_file,
        subprocess_salt2mu_res,
        subprocess_salt2mu_log,
        salt2mu_genpdf_grid=_CONFIG.salt2mu_genpdf_grid,
        debug=_WORKER_DEBUGFLAG,
        log_dir=log_dir,
        split_dist_par=_CONFIG.split_dist_par,
    )

    return connection


# =============================================================================
# LIKELIHOOD & PRIOR FUNCTIONS
# =============================================================================


def compute_and_sum_loglikelihoods(
    binned_dists: dict[str, list[NDArray]],
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
        binned_dists: Dictionary with (data, sim) tuples for each observable.
            Keys: 'c_hist', 'x1_hist', 'mures_high', 'mures_low',
            'rms_high', 'rms_low', 'nevt_high', 'nevt_low'.
            Each value is a tuple (real_data_array, sim_array).
        returnall: If True, return detailed components alongside the total.
        rms_weight: Weight factor for RMS terms in the likelihood.

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
    logger.debug(
        f"real sigint, sim sigint, real sigint error: "
        f"{_WORKER_REALDATA_SALT2MU_RESULTS['sigint']}, "
        f"{_WORKER_SALT2MU_CONNECTION.salt2mu_results['sigint']}, "
        f"{_WORKER_REALDATA_SALT2MU_RESULTS['siginterr']}"
    )

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

    # Salt parameters
    for k in ["c", "x1"]:
        res_key = f"{k}_hist"
        if res_key in binned_dists:
            datacount, simcount, poisson_err = norm_hist_to_data(*binned_dists[res_key])
            ll_dict[res_key] = -0.5 * np.sum((datacount - simcount) ** 2 / poisson_err**2)

            # DEBUG PURPOSE
            datacount_dict[res_key] = datacount
            simcount_dict[res_key] = simcount
            poisson_dict[res_key] = poisson_err
            logger.debug(
                f"   - {k}: ({datacount} (data) -  {simcount} (sim))**2 / {poisson_err}**2 = {ll_dict[k + '_hist']}"
            )

    for k in ["low", "high"]:
        mask = (binned_dists["nevt_" + k][0] > 0) & (binned_dists["nevt_" + k][1] > 0)

        # MURES #
        data_mures, sim_mures = binned_dists["mures_" + k]
        poisson_err_mures = np.sqrt(
            sum(
                (
                    rms**2 / nevt
                    for rms, nevt in zip(binned_dists["rms_" + k], binned_dists["nevt_" + k])
                )
            )
        )

        data_mures, sim_mures, poisson_err_mures = (
            data_mures[mask],
            sim_mures[mask],
            poisson_err_mures[mask],
        )

        ll_dict["mures_" + k] = -0.5 * np.sum((data_mures - sim_mures) ** 2 / poisson_err_mures**2)

        # DEBUG PURPOSE
        datacount_dict["mures_" + k] = data_mures
        simcount_dict["mures_" + k] = sim_mures
        poisson_dict["mures_" + k] = poisson_err_mures

        # RMS #
        data_rms, sim_rms = binned_dists["rms_" + k]
        poisson_err_rms = np.sqrt(
            sum(
                (
                    rms**2 / (2 * nevt)
                    for rms, nevt in zip(binned_dists["rms_" + k], binned_dists["nevt_" + k])
                )
            )
        )
        data_rms, sim_rms, poisson_err_rms = (
            data_rms[mask],
            sim_rms[mask],
            poisson_err_rms[mask],
        )

        ll_dict["rms_" + k] = (
            -0.5 * np.sum((data_rms - sim_rms) ** 2 / poisson_err_rms**2) * rms_weight
        )

        # DEBUG PURPOSE
        datacount_dict["rms_" + k] = data_rms
        simcount_dict["rms_" + k] = sim_rms
        poisson_dict["rms_" + k] = poisson_err_rms

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

    logger.debug(
        "Likelihood computation: \n"
        + "\n".join(
            [
                f" - LL {k} = {ll_dict[k]} for Ndata = {datacount_dict[k]} and Nsim={simcount_dict[k]}"
                for k in ll_dict
            ]
        )
    )
    return float(sum(ll_dict.values()))


def log_likelihood(
    theta_dic: dict[str, NDArray[np.float64]],
    returnall: bool = False,
    last: bool = False,
) -> float | tuple[dict, dict, dict, dict]:
    """
    Calculate log-likelihood for proposed parameter values.

    Core likelihood function for MCMC. For each parameter set:
    1. Writes PDF functions to file via connection.write_generic_PDF()
    2. Calls SALT2mu.exe to reweight simulation with those PDFs
    3. Parses binned output from SALT2mu (color histograms, MURES, RMS by mass)
    4. Compares reweighted simulation to real data

    Args:
        theta_dic: Dict mapping parameter names to their current values
            (after de-logging has been applied by log_probability).
        returnall: If True, return detailed likelihood components.
        last: If True, close the SALT2mu connection after this evaluation.

    Returns:
        Log-likelihood value (float).
        If returnall=True: tuple of (total_ll, ll_dict, datacount_dict, simcount_dict, poisson_dict).
        Returns -inf if MAXPROB > 1.001 (PDF probability hitting grid boundary).
    """

    # Run SALT2mu with these PDFs
    _WORKER_SALT2MU_CONNECTION.iterate(theta_dic, _CONFIG.fitted_params, last=last)

    if _WORKER_SALT2MU_CONNECTION.salt2mu_results["maxprob"] > 1.001:
        logger.warning(
            f"{_WORKER_SALT2MU_CONNECTION.salt2mu_results['maxprob']} MAXPROB > 1! "
            "Returning -np.inf"
        )
        return -np.inf

    # Build dictionary pairing data and simulation values
    binned_dists = {
        key: (
            _WORKER_REALDATA_SALT2MU_RESULTS["binned_dists"][key],
            _WORKER_SALT2MU_CONNECTION.salt2mu_results["binned_dists"][key],
        )
        for key in _WORKER_REALDATA_SALT2MU_RESULTS["binned_dists"].keys()
    }

    out_result = compute_and_sum_loglikelihoods(binned_dists, returnall=returnall)

    return out_result


def log_prior(theta_dic: dict[str, NDArray[np.float64]]) -> float:
    """
    Calculate log-prior probability for parameter values.

    Checks if all parameters are within their allowed bounds specified in
    config.parameter_initialization. Uses uniform (flat) priors within
    bounds, returning 0 (log(1)) if all parameters are valid or -inf if
    any parameter is outside its allowed range.

    Args:
        theta_dic: Dictionary mapping parameter names to values.

    Returns:
        0.0 if all parameters within bounds, -np.inf otherwise.
    """
    for p, val in theta_dic.items():
        bounds = _LIKELIHOOD_PARAMETERS["par_bounds"][p]
        if not (bounds[0] < val < bounds[1]):
            logger.debug(f"Prior on {[p]} is not in [{bounds[0]}, {bounds[1]}]")
            return -np.inf
    return 0.0


def log_probability(theta_dic: dict[str, np.float64] | list[float], **kwargs) -> float:
    """
    Calculate log-probability (posterior) for MCMC sampling.

    Combines log-prior and log-likelihood following Bayes' theorem.
    Must be called after _init_worker has set up the worker state.

    Args:
        theta_dic: Dict (or list/array accepted by emcee) of parameter values.
            Log-sampled parameters are exponentiated before prior/likelihood
            evaluation.

    Returns:
        Log-posterior probability (log_prior + log_likelihood).
    """
    logger.debug(
        f"\n\n#### COMPUTING LOGPROB ON ITERATION {_WORKER_SALT2MU_CONNECTION.iter} ####\n"
    )
    # De-log
    theta_dic = {
        k: np.exp(v) if _LIKELIHOOD_PARAMETERS["log_sampling"][k] else v
        for k, v in theta_dic.items()
    }

    logger.debug("   THETA: \n" + "\n".join([f"    -- {k} = {v}" for k, v in theta_dic.items()]))

    # Prior
    lp = log_prior(theta_dic)
    logger.debug(f"   LogPrior = {lp}")
    if not np.isfinite(lp):
        logger.debug("WARNING! We returned -inf from small parameters!")
        return -np.inf

    # Likelihood
    ll = log_likelihood(theta_dic, **kwargs)
    logger.debug(f"   LogLik = {ll}")

    logger.debug(f"\n#### END OF ITERATION  {_WORKER_SALT2MU_CONNECTION.iter} ####\n\n")

    # Increase iteration number
    _WORKER_SALT2MU_CONNECTION.iter += 1
    return lp + ll


# =============================================================================
# INITIALIZATION & WORKER SETUP
# =============================================================================


def _init_worker(
    config: Config,
    realdata_salt2mu_results: dict[str, Any],
    likelihood_parameters: dict[str, Any],
    debug: bool = False,
) -> None:
    """
    Initializer function for Pool workers.

    Sets up worker-local state by storing the appropriate connection for
    this worker based on its process identity. Called once per worker
    when the Pool is created.

    The ``debug`` flag controls both logging verbosity and SALT2mu
    behavior (optmask). To get DEBUG-level logging without changing
    SALT2mu behavior, set the logger level at the call site before or
    after calling this function (see --DEBUG_FULL).

    Args:
        config: Configuration object with parameters and paths.
        realdata_salt2mu_results: Dictionary containing real data fit results
            from SALT2mu (shared across workers).
        debug: If True, enable debug mode (short run, optmask=1,
            DEBUG-level logging).
    """
    global _WORKER_REALDATA_SALT2MU_RESULTS
    global _WORKER_SALT2MU_CONNECTION
    global _WORKER_DEBUGFLAG
    global _CONFIG
    global _WORKER_INDEX
    global _LIKELIHOOD_PARAMETERS

    _WORKER_DEBUGFLAG = debug
    _CONFIG = config
    _LIKELIHOOD_PARAMETERS = likelihood_parameters

    _WORKER_INDEX = get_worker_index()

    # Update logger level to match debug flag (worker may have been
    # initialised with setup_logging(debug=False) before the real flag
    # was known).
    if debug:
        logging.getLogger("dust2dusty").setLevel(logging.DEBUG)

    log_path = str(Path(config.OUTPUT_DIR) / "logs" / f"worker_{_WORKER_INDEX:02d}.log")
    add_file_handler(log_path)

    _WORKER_SALT2MU_CONNECTION = _init_salt2mu_worker_connection()
    _WORKER_REALDATA_SALT2MU_RESULTS = realdata_salt2mu_results

    logger.info(f"==== Worker {_WORKER_INDEX} INITIALIZED ====")
    logger.info(f"LIKELIHOOD PARAMETERS: {_LIKELIHOOD_PARAMETERS}")
    logger.info(f"DEBUG MODE: {_WORKER_DEBUGFLAG}")
    logger.info(f"Logger set to {log_path}")
    logger.info("============================================")


def cleanup_worker(_: Any = None) -> None:
    """
    Gracefully shut down the SALT2mu subprocess for this worker.

    Calls SALT2mu.close() to send the termination signal (-1) to the
    subprocess. Designed to be called via pool.map() so each worker
    cleans up its own connection.

    Args:
        _: Unused argument (required for pool.map compatibility).
    """
    global _WORKER_SALT2MU_CONNECTION

    if _WORKER_SALT2MU_CONNECTION is not None:
        logger.info(f"Worker {_WORKER_INDEX}: shutting down SALT2mu subprocess")
        _WORKER_SALT2MU_CONNECTION.close()
        _WORKER_SALT2MU_CONNECTION = None
