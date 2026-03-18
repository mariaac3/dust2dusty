"""
Utility functions for DUST2DUSTY that do not depend on global state.

These are pure functions that only use their input parameters,
making them easily testable and reusable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dust2dusty.cli import Config


_ARRAY_GENERATORS = {
    "arange": np.arange,
    "linspace": np.linspace,
}

# =============================================================================
# INIT TOOLS
# =============================================================================


def cmd_salt2mu_exe(config, data=False) -> str:
    """
    Build command string for SALT2mu.exe with subprocess file placeholders.

    Args:
        executable: Name of the executable (e.g., 'SALT2mu.exe').
        input_file: Path to the input file for SALT2mu.

    Returns:
        Command string with %s placeholders for subprocess files
        (mapsout, SALT2muout, log).
    """
    input_file = config.data_input if data else config.sim_input
    return f"{config._SALT2MU_EXE} {config.data_input} SUBPROCESS_FILES=%s,%s,%s "


def _init_salt2mu_realdata(
    config: Config,
    logger: logging.Logger,
    debug: bool = False,
    directory: str = "realdata_files",
) -> dict[str, Any]:
    """
    Initialize DUST2DUSTY by running SALT2mu on real data.

    Runs SALT2mu on real data to get baseline values for beta, betaerr,
    sigint, and siginterr that will be compared against in likelihood
    calculations. This establishes the "truth" values from observed data.

    Args:
        config: Configuration object containing paths and parameters.
        logger: Logger instance for output messages.
        debug: If True, use debug-mode filenames and pass debug flag to SALT2mu connection.

    Returns:
        Dictionary containing real data fit results with keys:
            - beta: Color-luminosity parameter
            - betaerr: Uncertainty on beta
            - sigint: Intrinsic scatter
            - siginterr: Uncertainty on sigint
            - bindf: Pandas DataFrame with binned statistics
            - alpha: SALT2 standardization parameter
            - alphaerr: Uncertainty on alpha
            - maxprob: Maximum probability ratio
    """
    index = "MASTER"

    if debug:
        index = "DEBUG"
    outdir = Path(config.OUTPUT_DIR)

    subprocess_log_data = outdir / f"{directory}/{index}_SALT2MU_SUBPROCESS_REALDATA_LOG.STDOUT"
    logger.debug(f"Create file: {subprocess_log_data.absolute()}")
    subprocess_log_data.touch()

    realdata_salt2mu_res = outdir / f"{directory}/REALDATA_SALT2MU_RES.DAT"
    logger.debug(f"Create file: {realdata_salt2mu_res.absolute()}")
    realdata_salt2mu_res.touch()

    # Generate output table specification (color bins x split parameter bins)
    arg_outtable = f"'c(6,-0.2:0.25)*HOST_LOGMASS(2,0:20)'"

    cmd = cmd_salt2mu_exe(config) + (f"SUBPROCESS_OUTPUT_TABLE={arg_outtable}")

    from dust2dusty.salt2mu import SALT2mu

    real_data = SALT2mu(
        cmd,
        outdir / f"{directory}/REALDATA_CROSSTALK_EMPTY.DAT",
        realdata_salt2mu_res,
        subprocess_log_data,
        is_realdata=True,
        debug=debug,
        split_dist_par=config.split_dist_par,
    )

    return real_data.salt2mu_results


def get_sampled_par_names_and_init(config) -> tuple[list[str], NDArray, NDArray, NDArray, NDArray]:
    sampled_par_names = []
    for p, pdic in config.fitted_params.items():
        if "splits" not in pdic:
            sampled_par_names.extend(
                [f"{p}_{dist_p}" for dist_p in config._DISTRIBUTION_PARAMETERS[pdic["dist"]]]
            )
        else:
            splits = pdic["splits"].keys()
            split_string = f"{p}_{{}}_" + "_".join([f"{split_p}_{{}}" for split_p in splits])
            split_states = []
            for r in range(len(splits) + 1):
                for low_vars in combinations(splits, r):
                    split_states.append(["low" if v in low_vars else "high" for v in splits])

            sampled_par_names.extend(
                [
                    split_string.format(dist_p, *split_st)
                    for split_st in split_states
                    for dist_p in config._DISTRIBUTION_PARAMETERS[pdic["dist"]]
                ]
            )
    p0_mu = np.array([config.parameter_inits[p]["p0"] for p in sampled_par_names])
    p0_std = np.array([config.parameter_inits[p]["p0_std"] for p in sampled_par_names])
    log_sampling = np.array(
        [True if "tau" in p or "std" in p else False for p in sampled_par_names], dtype=bool
    )

    # log sampling of variance
    p0_mu[log_sampling] = np.log(p0_mu[log_sampling])
    p0_std[log_sampling] = np.log(p0_std[log_sampling])

    par_bounds = np.array(
        [
            config.parameter_inits[p]["bounds"]
            if not log_sampling[i]
            else [-np.inf, config.parameter_inits[p]["bounds"][-1]]
            for i, p in enumerate(sampled_par_names)
        ]
    )

    return sampled_par_names, p0_mu, p0_std, par_bounds, log_sampling


# =============================================================================
# DATA PROCESSING
# =============================================================================


def binned_dist(
    salt2mu_res: pd.DataFrame, split_dist_par="HOST_LOGMASS"
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

    Returns:
        Dictionary with keys: 'mures_high', 'mures_low', 'rms_high', 'rms_low',
        'nevt_high', 'nevt_low', and optionally 'c_hist' and 'x1_hist' if those
        columns are present in the dataframe.
    """
    low_mask = salt2mu_res[f"ibin_{split_dist_par}"] == 0

    salt2mu_res_low = salt2mu_res[low_mask]
    salt2mu_res_high = salt2mu_res[~low_mask]

    low_NEVT = salt2mu_res_low["NEVT"].values
    high_NEVT = salt2mu_res_high["NEVT"].values

    mures_low = salt2mu_res_low["MURES_SUM"].values / low_NEVT
    mures_high = salt2mu_res_high["MURES_SUM"].values / high_NEVT

    rms_low = salt2mu_res_low.STD_ROBUST.values
    rms_high = salt2mu_res_high.STD_ROBUST.values

    par_pops = defaultdict(np.ndarray)
    for k in ["c", "x1"]:
        if f"ibin_{k}" in salt2mu_res.columns:
            par_pops[k + "_hist"] = salt2mu_res.groupby(f"ibin_{k}")["NEVT"].sum().values

    return {
        "mures_low": mures_low,
        "mures_high": mures_high,
        "rms_low": rms_low,
        "rms_high": rms_high,
        "nevt_low": low_NEVT,
        "nevt_high": high_NEVT,
        **par_pops,
    }


def norm_hist_to_data(
    datacount: NDArray[np.float64],
    simcount: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Normalize simulation histogram to match total counts in data.

    Scales simulation counts to have same total as data, computes Poisson
    errors, and masks bins where both data and sim are zero. This ensures
    fair comparison between data and simulation histograms regardless of
    total event counts.

    Args:
        datacount: Array of data histogram counts per bin.
        simcount: Array of simulation histogram counts per bin.

    Returns:
        Tuple of (datacount_masked, simcount_normalized, poisson_errors):
            - datacount_masked: Data counts with zero bins removed
            - simcount_normalized: Sim counts scaled by (datatot/simtot), zeros removed
            - poisson_err: Combined Poisson error per bin, zeros removed
    """
    datatot = np.sum(datacount)
    simtot = np.sum(simcount)

    norm = datatot / simtot

    ww = datacount != 0

    poisson_err = np.sqrt(datacount + simcount * norm**2)
    return datacount[ww], simcount[ww] * norm, poisson_err[ww]


def write_chain_to_text(
    chain: NDArray[np.float64],
    log_prob: NDArray[np.float64],
    param_names: list[str],
    filepath: str,
) -> None:
    """
    Write MCMC chain and log-probability to a tab-separated text file.

    Each row represents one walker at one iteration. The file includes a
    header line with column names prefixed by '#'.

    Args:
        chain: Chain array of shape (n_iterations, nwalkers, ndim).
        log_prob: Log-probability array of shape (n_iterations, nwalkers).
        param_names: List of parameter names (length = ndim).
        filepath: Output file path.
    """
    n_iters, n_walk, _ = chain.shape
    with open(filepath, "w") as f:
        header_cols = ["iteration", "walker"] + param_names + ["log_prob"]
        f.write("# " + "\t".join(header_cols) + "\n")
        for i in range(n_iters):
            for w in range(n_walk):
                row = [str(i), str(w)]
                row += [f"{val:.8e}" for val in chain[i, w, :]]
                row.append(f"{log_prob[i, w]:.8e}")
                f.write("\t".join(row) + "\n")


__dust2dust_str__ = """
    ██████╗ ██╗   ██╗███████╗████████╗██████╗ ██████╗ ██╗   ██╗███████╗████████╗
    ██╔══██╗██║   ██║██╔════╝╚══██╔══╝╚════██╗██╔══██╗██║   ██║██╔════╝╚══██╔══╝
    ██║  ██║██║   ██║███████╗   ██║    █████╔╝██║  ██║██║   ██║███████╗   ██║
    ██║  ██║██║   ██║╚════██║   ██║   ██╔═══╝ ██║  ██║██║   ██║╚════██║   ██║
    ██████╔╝╚██████╔╝███████║   ██║   ███████╗██████╔╝╚██████╔╝███████║   ██║
    ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝   ╚═╝
    """
