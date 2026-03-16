"""
Utility functions for DUST2DUSTY that do not depend on global state.

These are pure functions that only use their input parameters,
making them easily testable and reusable.
"""

from __future__ import annotations

import logging
import os
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dust2dusty.cli import Config

# Constants
JOBNAME_SALT2MU: str = "SALT2mu.exe"

_ARRAY_GENERATORS = {
    "arange": np.arange,
    "linspace": np.linspace,
}


def generate_split_array(spec: dict[str, Any]) -> NDArray[np.float64]:
    """Generate a numpy array from a splitarr config entry.

    Args:
        spec: Dict with keys 'method' (arange or linspace) and 'args' (list of numbers).

    Returns:
        The generated numpy array.
    """
    return _ARRAY_GENERATORS[spec["method"]](*spec["args"])


def cmd_exe(executable: str, input_file: str) -> str:
    """
    Build command string for SALT2mu.exe with subprocess file placeholders.

    Args:
        executable: Name of the executable (e.g., 'SALT2mu.exe').
        input_file: Path to the input file for SALT2mu.

    Returns:
        Command string with %s placeholders for subprocess files
        (mapsout, SALT2muout, log).
    """
    return f"{executable} {input_file} SUBPROCESS_FILES=%s,%s,%s "


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
    outdir = Path(config.outdir)

    subprocess_log_data = outdir / f"{directory}/{index}_SALT2MU_SUBPROCESS_REALDATA_LOG.STDOUT"
    logger.debug(f"Create file: {subprocess_log_data.absolute()}")
    subprocess_log_data.touch()

    realdata_salt2mu_res = outdir / f"{directory}/REALDATA_SALT2MU_RES.DAT"
    logger.debug(f"Create file: {realdata_salt2mu_res.absolute()}")
    realdata_salt2mu_res.touch()

    # Generate output table specification (color bins x split parameter bins)
    arg_outtable = f"'c(6,-0.2:0.25)*{config.SPLIT_PARAMETER_FORMATS[config.splitparam]}'"

    cmd = cmd_exe(JOBNAME_SALT2MU, config.data_input) + (
        f"SUBPROCESS_OUTPUT_TABLE={arg_outtable} debug_flag=930"
    )

    from dust2dusty.salt2mu import SALT2mu

    real_data = SALT2mu(
        cmd,
        outdir / f"{directory}/REALDATA_CROSSTALK_EMPTY.DAT",
        realdata_salt2mu_res,
        subprocess_log_data,
        is_realdata=True,
        debug=debug,
    )

    return real_data.salt2mu_results


def get_sampled_par_names_and_init(config) -> list[str]:
    sampled_par_names = []
    for p, pdic in config.fitted_params.items():
        if "splits" not in pdic:
            sampled_par_names.extend(
                [f"{p}_{dist_p}" for dist_p in config.DISTRIBUTION_PARAMETERS[pdic["dist"]]]
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
                    for dist_p in config.DISTRIBUTION_PARAMETERS[pdic["dist"]]
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
            else [-np.inf, np.log(config.parameter_inits[p]["bounds"][-1])]
            for i, p in enumerate(sampled_par_names)
        ]
    )

    return sampled_par_names, p0_mu, p0_std, par_bounds, log_sampling


def input_cleaner(config) -> tuple[NDArray[np.float64], int, int]:
    """
    Initialize MCMC walker starting positions with appropriate constraints.

    Generates initial walker positions for emcee sampler, ensuring all
    parameters start within their valid bounds and with appropriate spreads.

    Args:
        fitted_params: List of parameter names to fit (e.g., ['c', 'RV', 'EBV']).
        param_dists: Maps parameter to distribution shape.
        splitdict: Nested dict defining parameter splits.
        distribution_parameters: Dict mapping distribution names to parameter names.
        parameter_initialization: Dictionary containing initialization info for
            each expanded parameter. Each entry is a dict with keys:
            start, stdev, require_positive, bounds (a [lower, upper] list).
        parameter_overrides: Dictionary of parameters to fix (not fit).
        walkfactor: Multiplier for number of walkers (nwalkers = ndim * walkfactor).

    Returns:
        Tuple of (pos, nwalkers, ndim) where:
            - pos: Array of shape (nwalkers, ndim) with initial walker positions
            - nwalkers: Number of MCMC walkers
            - ndim: Number of dimensions (parameters)
    """
    plist = pconv(fitted_params, param_dists, splitdict, distribution_parameters)
    nwalkers = len(plist) * walkfactor
    for element in parameter_overrides.keys():
        plist.remove(element)
    pos = np.abs(0.1 * np.random.randn(nwalkers, len(plist)))
    for entry in range(len(plist)):
        newpos_param = parameter_initialization[plist[entry]]
        pos[:, entry] = np.random.normal(
            newpos_param["start"], newpos_param["stdev"], len(pos[:, entry])
        )
        if newpos_param["require_positive"]:
            pos[:, entry] = np.abs(pos[:, entry])
        while any(ele < newpos_param["bounds"][0] for ele in pos[:, entry]) or any(
            ele > newpos_param["bounds"][1] for ele in pos[:, entry]
        ):
            pos[:, entry] = np.random.normal(
                newpos_param["start"], newpos_param["stdev"], len(pos[:, entry])
            )
            if newpos_param["require_positive"]:
                pos[:, entry] = np.abs(pos[:, entry])
    return pos, nwalkers, len(plist)


def subprocess_to_snana(outdir: str, snana_mapping: dict[str, str]) -> str:
    """
    Convert GENPDF file from SUBPROCESS format to SNANA-compatible format.

    Reads GENPDF.DAT file, removes the first line (header), and replaces
    variable names from subprocess format (e.g., 'SIM_c', 'SIM_RV') to SNANA
    format (e.g., 'SALT2c', 'RV') so the file can be used directly in SNANA
    simulations.

    Args:
        outdir: Output directory containing GENPDF.DAT (should end with '/').
        snana_mapping: Dictionary mapping subprocess names to SNANA names.
            Example: {'SIM_c': 'SALT2c', 'SIM_RV': 'RV', 'HOST_LOGMASS': 'LOGMASS'}.

    Returns:
        'Done' upon successful completion.

    Side Effects:
        Modifies GENPDF.DAT file in place:
        - Removes first line
        - Converts all variable names to SNANA format
    """
    filein = outdir + "GENPDF.DAT"
    with open(filein) as f:
        lines = f.readlines()
    del lines[0]
    os.remove(filein)
    with open(filein, "w+") as f:
        for line in lines:
            f.write(line)
    with open(filein) as f:
        filedata = f.read()
    for key in snana_mapping.keys():
        if key in filedata:
            filedata = filedata.replace(key, snana_mapping[key])
    os.remove(filein)
    with open(filein, "w") as f:
        f.write(filedata)
    return "Done"


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
